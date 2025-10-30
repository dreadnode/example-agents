<?php

/**
 * Visibility Settings Module
 * Settings > Widget Options :: Pages Visibility
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Create Card Module for Pages Visibility Options
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */
if (!function_exists('widgetopts_settings_visibility')) :
	function widgetopts_settings_visibility()
	{
		global $widget_options; ?>
		<li class="widgetopts-module-card <?php echo ($widget_options['visibility'] == 'activate') ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-visibility" data-module-id="visibility">
			<div class="widgetopts-module-card-content">
				<h2><?php _e('Pages Visibility', 'widget-options'); ?></h2>
				<p class="widgetopts-module-desc">
					<?php _e('Easily restrict any widgets visibility on specific WordPress pages.', 'widget-options'); ?>
				</p>
				<div class="widgetopts-module-actions hide-if-no-js">
					<?php if ($widget_options['visibility'] == 'activate') { ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e('Configure Settings', 'widget-options'); ?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e('Disable', 'widget-options'); ?></button>
					<?php } else { ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e('Learn More', 'widget-options'); ?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e('Enable', 'widget-options'); ?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start($widget_options['visibility']); ?>
			<span class="dashicons widgetopts-dashicons dashicons-visibility"></span>
			<h3 class="widgetopts-modal-header"><?php _e('Pages Visibility', 'widget-options'); ?></h3>
			<p>
				<?php _e('Visibility tab allows you to completely control each widgets visibility and restrict them on any WordPress pages. You can turn on/off the underlying tabs for post types, taxonomies and miscellanous options using the options below when this feature is enabled.', 'widget-options'); ?>
			</p>
			<table class="form-table widgetopts-settings-section">
				<tr>
					<th scope="row">
						<label for="widgetopts-visibility-misc"><?php _e('Pages Tab', 'widget-options'); ?></label>
					</th>
					<td>
						<input type="checkbox" id="widgetopts-visibility-misc" name="visibility[misc]" <?php echo (isset($widget_options['settings']['visibility'])) ? widgetopts_is_checked($widget_options['settings']['visibility'], 'misc') : ''; ?> value="1" />
						<label for="widgetopts-visibility-misc"><?php _e('Enable Miscellaneous Options', 'widget-options'); ?></label>
						<p class="description">
							<?php _e('Restrict widgets or blocks visibility on WordPress new added page as well as default pages such as home page, blog page, 404, search, etc.', 'widget-options'); ?>
						</p>
					</td>
				</tr>
				<tr>
					<th scope="row">
						<label for="widgetopts-visibility-post_types"><?php _e('Post Types Tab', 'widget-options'); ?></label>
					</th>
					<td>
						<input type="checkbox" id="widgetopts-visibility-post_types" name="visibility[post_type]" <?php echo (isset($widget_options['settings']['visibility'])) ? widgetopts_is_checked($widget_options['settings']['visibility'], 'post_type') : ''; ?> value="1" />
						<label for="widgetopts-visibility-post_types"><?php _e('Enable Post Types Restriction', 'widget-options'); ?></label>
						<p class="description">
							<?php _e('This feature will allow visibility restriction of every widgets or blocks per post types and per pages.', 'widget-options'); ?>
						</p>
					</td>
				</tr>
				<tr>
					<th scope="row">
						<label for="widgetopts-visibility-taxonomies"><?php _e('Taxonomies Tab', 'widget-options'); ?></label>
					</th>
					<td>
						<input type="checkbox" id="widgetopts-visibility-taxonomies" name="visibility[taxonomies]" <?php echo (isset($widget_options['settings']['visibility'])) ? widgetopts_is_checked($widget_options['settings']['visibility'], 'taxonomies') : ''; ?> value="1" />
						<label for="widgetopts-visibility-taxonomies"><?php _e('Enable Taxonomies Restriction', 'widget-options'); ?></label>
						<p class="description">
							<?php _e('This tab option will allow you to control visibility via taxonomy and terms archive pages.', 'widget-options'); ?>
						</p>
					</td>
				</tr>

			</table>
			<?php widgetopts_modal_end($widget_options['visibility']); ?>
		</li>
<?php
	}
	add_action('widgetopts_module_cards', 'widgetopts_settings_visibility', 12);
endif;
?>