<?php
/**
 * Shortcodes Settings Module
 * Settings > Widget Options :: Shortcodes
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Shortcodes Options
 *
 * @since 3.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_shortcodes' ) ):
	function widgetopts_settings_shortcodes(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="shortcodes">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/sidebar-widgets-shortcodes/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Shortcodes', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Display any sidebars and widgets anywhere using shortcodes.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_shortcodes', 170 );
endif;
?>
