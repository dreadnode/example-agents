<?php
/**
 * Alignment Settings Module
 * Settings > Widget Options :: Custom Alignment
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Custom Alignment Options
 *
 * @since 3.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_permissions' ) ):
	function widgetopts_settings_permissions(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="permission">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Permission', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Hide widget options tabs below each widgets to selected user roles.', 'widget-options' );?>
				</p>
			</div>
		</li>

	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_permissions', 160 );
endif;
?>
